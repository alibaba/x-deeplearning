<?php

namespace Google\Protobuf\Internal;

use Google\Protobuf\EnumValueDescriptor;

class EnumDescriptor
{
    use HasPublicDescriptorTrait;

    private $klass;
    private $full_name;
    private $value;
    private $name_to_value;
    private $value_descriptor = [];

    public function __construct()
    {
        $this->public_desc = new \Google\Protobuf\EnumDescriptor($this);
    }

    public function setFullName($full_name)
    {
        $this->full_name = $full_name;
    }

    public function getFullName()
    {
        return $this->full_name;
    }

    public function addValue($number, $value)
    {
        $this->value[$number] = $value;
        $this->name_to_value[$value->getName()] = $value;
        $this->value_descriptor[] = new EnumValueDescriptor($value->getName(), $number);
    }

    public function getValueByNumber($number)
    {
        return $this->value[$number];
    }

    public function getValueByName($name)
    {
        return $this->name_to_value[$name];
    }

    public function getValueDescriptorByIndex($index)
    {
        return $this->value_descriptor[$index];
    }

    public function getValueCount()
    {
        return count($this->value);
    }

    public function setClass($klass)
    {
        $this->klass = $klass;
    }

    public function getClass()
    {
        return $this->klass;
    }

    public static function buildFromProto($proto, $file_proto, $containing)
    {
        $desc = new EnumDescriptor();

        $enum_name_without_package  = "";
        $classname = "";
        $fullname = "";
        GPBUtil::getFullClassName(
            $proto,
            $containing,
            $file_proto,
            $enum_name_without_package,
            $classname,
            $fullname);
        $desc->setFullName($fullname);
        $desc->setClass($classname);
        $values = $proto->getValue();
        foreach ($values as $value) {
            $desc->addValue($value->getNumber(), $value);
        }

        return $desc;
    }
}
